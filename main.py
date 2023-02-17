import pygame as p
from copy import deepcopy
import numpy as np
from numba import jit

H = 600
W = 800
disp = p.display.set_mode((W, H))
FPS = 60
white = (255, 255, 255)
black = (0, 0, 0)

p.font.init()
clock = p.time.Clock()
p.display.set_caption('3d demo')
font = p.font.SysFont('ubuntu mono', 20)


def createMat():
    return np.zeros((4, 4), dtype=np.float32)


def createVector(x=0, y=0, z=0, w=1):
    return np.array([x, y, z, w], dtype=np.float32)


def createTriangle():
    return np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.float64)


def createIdentity():
    return np.eye(4, 4)


def matrix_makeProj():
    a = H / W
    far = 1000  # stuff for 3D
    near = 0.1  # like most of the code)
    FieldOfView = 90
    angle = (FieldOfView * 0.5) / 180 * np.pi  # 180 times pi is to convert to radians
    FovRad = 1 / np.tan(angle)

    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0][0] = a * FovRad
    mat[1][1] = FovRad
    mat[2][2] = far / (far - near)
    mat[3][2] = (-far * near) / (far - near)
    mat[2][3] = 1
    return mat


def matrix_makeTranslation(x, y, z):
    mat = np.eye(4, 4)
    mat[3][0] = x
    mat[3][1] = y
    mat[3][2] = z
    return mat


def FillTriangle(tri, dp, color):
    dp = min(dp, 1)
    color = [color[0] * dp, color[1] * dp, color[2] * dp]
    points = [(tri[0][0], tri[0][1]), (tri[1][0], tri[1][1]), (tri[2][0], tri[2][1])]
    p.draw.polygon(disp, color, points)


@jit(nopython=True)
def matrixVector_Mul(mat, v):
    vect = np.zeros(4, dtype=np.float32)
    vect[0] = v[0] * mat[0][0] + v[1] * mat[1][0] + v[2] * mat[2][0] + v[3] * mat[3][0]
    vect[1] = v[0] * mat[0][1] + v[1] * mat[1][1] + v[2] * mat[2][1] + v[3] * mat[3][1]
    vect[2] = v[0] * mat[0][2] + v[1] * mat[1][2] + v[2] * mat[2][2] + v[3] * mat[3][2]
    vect[3] = v[0] * mat[0][3] + v[1] * mat[1][3] + v[2] * mat[2][3] + v[3] * mat[3][3]
    return vect


def matrixMatrix_Mul(mat1, mat2):
    return np.matmul(mat1, mat2)


def get_mesh(filename):
    #filename = input('input obj name:')
    try:
        f = open(f'{filename}.obj', 'r')
    except:
        print("obj wasn't converted, perhaps name error")
        exit()
    else:
        mesh = list()
        lines = np.array(f.readlines())
        vert = list()

        for i in lines:
            i = i.split()
            try:
                if i[0] == 'v':
                    i = i[1:]
                    for g in range(len(i)):
                        i[g] = float(i[g])
                    vert.append(i)

                if i[0] == 'f':
                    i = i[1:]
                    for g in range(len(i)):
                        i[g] = int(i[g])
                    mesh.append([vert[i[0] - 1], vert[i[1] - 1], vert[i[2] - 1]])
            except IndexError:
                pass
        del vert
        del lines
        return np.array(mesh, dtype=np.float64)

@jit(nopython=True)
def Sort(tri):
    return (tri[0][2] + tri[1][2] + tri[2][2]) / 3


@jit(nopython=True)
def vector_add(v1, v2):
    vect = np.zeros(4, dtype=np.float32)
    vect[0] = v1[0] + v2[0]
    vect[1] = v1[1] + v2[1]
    vect[2] = v1[2] + v2[2]
    vect[3] = 1
    return vect


@jit(nopython=True)
def vector_sub(v1, v2):
    vect = np.zeros(4, dtype=np.float32)
    vect[0] = v1[0] - v2[0]
    vect[1] = v1[1] - v2[1]
    vect[2] = v1[2] - v2[2]
    vect[3] = 1
    return vect


@jit(nopython=True)
def vector_mul(v, k):
    v[0] = v[0] * k
    v[1] = v[1] * k
    v[2] = v[2] * k
    return v


@jit(nopython=True)
def vector_div(v, k):
    if k != 0:
        v[0] = v[0] / k
        v[1] = v[1] / k
        v[2] = v[2] / k
    return v


@jit(nopython=True)
def vector_dotProduct(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


@jit(nopython=True)
def Vector_len(v):
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@jit(nopython=True)
def vector_norm(v):
    length = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length != 0:
        v[0] = v[0] / length
        v[1] = v[1] / length
        v[2] = v[2] / length
    return v


@jit(nopython=True)
def vector_crossProd(v1, v2):
    vect = np.zeros(4, dtype=np.float32)
    vect[0] = v1[1] * v2[2] - v1[2] * v2[1]
    vect[1] = v1[2] * v2[0] - v1[0] * v2[2]
    vect[2] = v1[0] * v2[1] - v1[1] * v2[0]
    vect[3] = 1
    return vect


@jit(nopython=True)
def matrixRotationZ(angle):
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0][0] = np.cos(angle)
    mat[0][1] = np.sin(angle)
    mat[1][0] = -np.sin(angle)
    mat[1][1] = np.cos(angle)
    mat[2][2] = 1
    mat[3][3] = 1
    return mat


@jit(nopython=True)
def matrixRotationX(angle):
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0][0] = 1
    mat[1][1] = np.cos(angle)
    mat[1][2] = np.sin(angle)
    mat[2][1] = -np.sin(angle)
    mat[2][2] = np.cos(angle)
    mat[3][3] = 1
    return mat


@jit(nopython=True)
def matrixRotationY(angle):
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0][0] = np.cos(angle)
    mat[0][2] = np.sin(angle)
    mat[2][0] = -np.sin(angle)
    mat[1][1] = 1
    mat[2][2] = np.cos(angle)
    mat[3][3] = 1
    return mat


def matrix_Inverse(m):
    mat = np.zeros((4, 4))
    mat[0][0] = m[0][0]
    mat[0][1] = m[1][0]
    mat[0][2] = m[2][0]
    

    mat[1][0] = m[0][1]
    mat[1][1] = m[1][1]
    mat[1][2] = m[2][1]
    
    
    mat[2][0] = m[0][2]
    mat[2][1] = m[1][2]
    mat[2][2] = m[2][2]
    

    mat[3][0] = -(m[3][0] * mat[0][0] + m[3][1] * mat[1][0] + m[3][2] * mat[2][0])
    mat[3][1] = -(m[3][0] * mat[0][1] + m[3][1] * mat[1][1] + m[3][2] * mat[2][1])
    mat[3][2] = -(m[3][0] * mat[0][2] + m[3][1] * mat[1][2] + m[3][2] * mat[2][2])
    mat[3][3] = 1

    return mat


def matrix_PointAt(pos, target, up):

    newForward = createVector()
    newForward = vector_sub(target, pos)
    newForward = vector_norm(newForward)

    a = createVector()
    a = vector_mul(newForward, vector_dotProduct(up, newForward))

    newUp = createVector()
    newUp = vector_sub(up, a)
    newUp = vector_norm(newUp)

    newRight = createVector()
    newRight = vector_crossProd(newUp, newForward)

    mat = np.zeros((4, 4))

    mat[0][0] = newRight[0]
    mat[0][1] = newRight[1]
    mat[0][2] = newRight[2]
    

    mat[1][0] = newUp[0]
    mat[1][1] = newUp[1]
    mat[1][2] = newUp[2]
    

    mat[2][0] = newForward[0]
    mat[2][1] = newForward[1]
    mat[2][2] = newForward[2]

    mat[3][0] = pos[0]
    mat[3][1] = pos[1]
    mat[3][2] = pos[2]
    mat[3][3] = 1

    return mat


def Vector_IntersectPlane(plane_p,plane_n,lineStart, lineEnd):
    plane_n = vector_norm(plane_n)
    plane_d = -vector_dotProduct(lineStart, plane_n) #perhaps will be error, then just create every vector separatly
    ab = vector_dotProduct(lineStart, plane_n)
    bd = vector_dotProduct(lineEnd, plane_n)
    t = (-plane_d - ab) / (bd - ab)
    lineStartToEnd = vector_sub(lineEnd, lineStart)
    lineToIntersect = vector_mul(lineStartToEnd, t)
    return vector_add(lineStart, lineToIntersect)

def Triangle_ClipAgainstPlane(plane_p, plane_n, in_tri, out_tri1, out_tri2):
    plane_n = vector_norm(plane_n)
    def dist(p):
        n = vector_norm(p)
        return (plane_n[0] * p[0] + plane_n[1] * p[1] + plane_n[2] * p[2] - vector_dotProduct(plane_n, plane_p))
    
    nInsidePointCount = 0
    nOutsidePointCount = 0
    inside_points = [createVector(),createVector(),createVector()]
    outside_points = [createVector(),createVector(),createVector()]

    d0 = dist(in_tri[0])
    d1 = dist(in_tri[1])
    d2 = dist(in_tri[2])
    if d0 >= 0: 
        inside_points[nInsidePointCount] = in_tri[0]
        nInsidePointCount += 1
    else:
        outside_points[nOutsidePointCount] = in_tri[0]
        nOutsidePointCount += 1
    if d1 >= 0: 
        inside_points[nInsidePointCount] = in_tri[1]
        nInsidePointCount += 1
    else:
        outside_points[nOutsidePointCount] = in_tri[1]
        nOutsidePointCount += 1
    if d2 >= 0: 
        inside_points[nInsidePointCount] = in_tri[2]
        nInsidePointCount += 1
    else:
        outside_points[nOutsidePointCount] = in_tri[2]
        nOutsidePointCount += 1
    

    if nInsidePointCount == 0:
        return 0
    if nInsidePointCount == 3:
        out_tri1 = in_tri
        return 1
    if  nInsidePointCount == 1 and nOutsidePointCount == 2:
        out_tri1 = in_tri
        out_tri1[0] = inside_points[0]
        out_tri1[1] = Vector_IntersectPlane(plane_p, plane_n, inside_points[0], outside_points[0])
        out_tri1[2] = Vector_IntersectPlane(plane_p, plane_n, inside_points[0], outside_points[1])
        return 1
    if nInsidePointCount == 2 and nOutsidePointCount == 1:
        out_tri1 = in_tri
        out_tri2 = in_tri

        out_tri1[0] = inside_points[0]
        out_tri1[1] = inside_points[1]
        out_tri1[2] = Vector_IntersectPlane(plane_p, plane_n, inside_points[0], outside_points[0])

        out_tri2[0] = inside_points[1]
        out_tri2[1] = out_tri1[2]
        out_tri2[2] = Vector_IntersectPlane(plane_p, plane_n, inside_points[1], outside_points[0])

        return 2

def main():
    fTheta = 0
    fYaw = 0
    mesh = get_mesh('axis')

    vCamera = createVector()
    
    vLookDir = createVector(x=0,y=1,z=1)

    triProj = createTriangle()

    matProj = matrix_makeProj()

    trianglesToRaster = list()
    append = trianglesToRaster.append

    while True:
        for event in p.event.get():
            if event.type == p.QUIT:
                exit()
            if event.type == p.KEYDOWN:
                
                vForward = vector_mul(vLookDir, 0.08*clock.get_time())

                if event.key == p.K_DOWN:
                    vCamera[1] -= 0.02 * clock.get_time()
                if event.key == p.K_UP:
                    vCamera[1] += 0.02 * clock.get_time()
                if event.key == p.K_RIGHT:
                    vCamera[0] += 0.02 * clock.get_time()
                if event.key == p.K_LEFT:
                    vCamera[0] -= 0.02 * clock.get_time()


                if event.key == ord('s'):
                    vCamera = vector_sub(vCamera, vForward)
                if event.key == ord('w'):
                    vCamera = vector_add(vCamera, vForward)
                if event.key == ord('d'):
                    fYaw += 0.02 * clock.get_time()
                if event.key == ord('a'):
                    fYaw -= 0.02 * clock.get_time()
                



        disp.fill(black)
        
        matTrans = matrix_makeTranslation(0, 5, 5)

        matWorld = createIdentity()
        matWorld = np.matmul(matWorld, matTrans)

        vTarget = createVector(x=0,y=1,z=1)
        vUp = createVector(x=0,y=1,z=0)
        matCameraRot = matrixRotationY(fYaw)
        vLookDir = matrixVector_Mul(matCameraRot, vTarget)


        vTarget = vector_add(vCamera, vLookDir)
        
        matCamera = matrix_PointAt(vCamera, vTarget, vUp)


        matVeiw = matrix_Inverse(matCamera)




        for tri in mesh:

            triViewed = createTriangle()

            tri0 = createVector(x=tri[0][0], y=tri[0][1], z=tri[0][2])
            tri1 = createVector(x=tri[1][0], y=tri[1][1], z=tri[1][2])
            tri2 = createVector(x=tri[2][0], y=tri[2][1], z=tri[2][2])

            triTransformed = createTriangle()


            triTransformed[0] = matrixVector_Mul(matWorld, tri0)
            triTransformed[1] = matrixVector_Mul(matWorld, tri1)
            triTransformed[2] = matrixVector_Mul(matWorld, tri2)

            line1 = vector_sub(triTransformed[1], triTransformed[0])
            line2 = vector_sub(triTransformed[2], triTransformed[0])

            normal = vector_norm(vector_crossProd(line1, line2))

            vCameraRay = vector_sub(triTransformed[0], vCamera)

            if vector_dotProduct(vCameraRay, normal) < 0:
                light_direction = createVector(x=0, y=1, z=-1)
                light_direction = vector_norm(light_direction)

                dp = max(0.1, vector_dotProduct(light_direction, normal))

                triViewed[0] = matrixVector_Mul(matVeiw, triTransformed[0])
                triViewed[1] = matrixVector_Mul(matVeiw, triTransformed[1])
                triViewed[2] = matrixVector_Mul(matVeiw, triTransformed[2])

                #3D -> 2D
                triProj[0] = matrixVector_Mul(matProj, triViewed[0])
                triProj[1] = matrixVector_Mul(matProj, triViewed[1])
                triProj[2] = matrixVector_Mul(matProj, triViewed[2])

                #Scaling
                triProj[0] = vector_div(triProj[0], triProj[0][3])
                triProj[1] = vector_div(triProj[1], triProj[1][3])
                triProj[2] = vector_div(triProj[2], triProj[2][3])

                vOffset = createVector(x=1, y=1, z=0)

                triProj[0] = vector_add(triProj[0], vOffset)
                triProj[1] = vector_add(triProj[1], vOffset)
                triProj[2] = vector_add(triProj[2], vOffset)

                triProj[0][0] *= 0.5 * W
                triProj[0][1] *= 0.5 * H
                triProj[1][0] *= 0.5 * W
                triProj[1][1] *= 0.5 * H
                triProj[2][0] *= 0.5 * W
                triProj[2][1] *= 0.5 * H

                triProj[3][0] = dp

                append(deepcopy(triProj))

        trianglesToRaster.sort(reverse=True, key=Sort)

        for tr in trianglesToRaster:
            FillTriangle(tr, tr[3][0], [255, 255, 255])

        trianglesToRaster.clear()

        #printing fps
        text = font.render(str(clock.get_fps())[:5], True, white)
        place = text.get_rect(center=(20, 20))
        disp.blit(text, place)
        p.display.update()
        clock.tick(FPS)


if __name__ == '__main__':
    main()

